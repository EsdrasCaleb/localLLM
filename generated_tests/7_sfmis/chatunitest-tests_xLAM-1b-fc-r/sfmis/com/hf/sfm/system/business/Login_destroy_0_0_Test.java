package com.hf.sfm.system.business;

import com.hf.sfm.system.business.Login;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
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
class Login_destroy_0_0_Test {

    @Mock
    Login login;

    @InjectMocks
    Login_destroy_0_0_Test login_destroy_0_0_Test;

    @Test
    void destroyTest() {
        // Arrange
        // Act
        login_destroy_0_0_Test.login.destroy();
        // Assert
        verify(login, times(1)).destroy();
    }
}
