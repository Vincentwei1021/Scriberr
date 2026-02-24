package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"scriberr/internal/auth"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestAuthMiddleware_DisableAuth_AllowsWithoutCredentials(t *testing.T) {
	t.Setenv("DISABLE_AUTH", "true")
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.GET("/protected", AuthMiddleware(auth.NewAuthService("test-secret")), func(c *gin.Context) {
		authType, _ := c.Get("auth_type")
		userID, _ := c.Get("user_id")
		username, _ := c.Get("username")
		c.JSON(http.StatusOK, gin.H{
			"auth_type": authType,
			"user_id":   userID,
			"username":  username,
		})
	})

	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"auth_type":"disabled"`)
	assert.Contains(t, w.Body.String(), `"username":"local-user"`)
}

func TestJWTOnlyMiddleware_DisableAuth_AllowsWithoutAuthorizationHeader(t *testing.T) {
	t.Setenv("DISABLE_AUTH", "true")
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.GET("/jwt-protected", JWTOnlyMiddleware(auth.NewAuthService("test-secret")), func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	req := httptest.NewRequest(http.MethodGet, "/jwt-protected", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"ok":true`)
}

