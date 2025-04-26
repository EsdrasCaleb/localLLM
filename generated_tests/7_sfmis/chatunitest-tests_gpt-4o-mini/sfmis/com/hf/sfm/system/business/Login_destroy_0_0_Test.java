package com.hf.sfm.system.business;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.IOException;
import javax.servlet.ServletException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Iterator;
import java.util.List;
import javax.servlet.http.HttpServlet;
import org.hibernate.Session;
import com.hf.sfm.util.DaoFactory;
import com.hf.sfm.util.HibernateSessionFactory;

@ExtendWith(MockitoExtension.class)
public class Login_destroy_0_0_Test {

    private Login login;

    @BeforeEach
    public void setUp() {
        login = Mockito.spy(new Login());
    }

    @Test
    public void testDestroy() throws IOException, ServletException {
        // Arrange
        HttpServletRequest request = mock(HttpServletRequest.class);
        HttpServletResponse response = mock(HttpServletResponse.class);
        HttpSession session = mock(HttpSession.class);
        when(request.getSession()).thenReturn(session);
        // Act
        // Invoke the destroy method using reflection
        try {
            java.lang.reflect.Method method = Login.class.getDeclaredMethod("destroy", HttpServletRequest.class, HttpServletResponse.class);
            method.setAccessible(true);
            method.invoke(login, request, response);
        } catch (Exception e) {
            fail("Method invocation failed: " + e.getMessage());
        }
        // Assert
        verify(session).invalidate();
    }
}
