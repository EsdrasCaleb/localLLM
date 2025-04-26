package com.hf.sfm.system.business;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.HashMap;
import java.util.Map;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import org.hibernate.Session;
import com.hf.sfm.util.DaoFactory;
import com.hf.sfm.util.HibernateSessionFactory;

@ExtendWith(MockitoExtension.class)
class Login_doGet_1_2_Test {

    @Mock
    private HttpServletRequest request;

    @Mock
    private HttpServletResponse response;

    @InjectMocks
    private Login login;

    @Test
    void testDoGet() {
        // Arrange
        Map<String, String> headers = new HashMap<>();
        headers.put("Authorization", "Bearer validToken");
        // Act
        when(request.getHeader("Authorization")).thenReturn("Bearer invalidToken");
        when(request.getHeader("Authorization")).thenReturn("Bearer validToken");
        when(request.getHeader("Authorization")).thenReturn(null);
        // Assert
        assertThrows(IllegalArgumentException.class, () -> login.doGet(request, response));
    }
}
