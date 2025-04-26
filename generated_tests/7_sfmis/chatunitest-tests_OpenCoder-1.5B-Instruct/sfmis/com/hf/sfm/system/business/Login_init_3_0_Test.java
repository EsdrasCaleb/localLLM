package com.hf.sfm.system.business;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
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

class Login_init_3_0_Test {

    private Login login;

    private HttpServletRequest request;

    private HttpServletResponse response;

    private Session session;

    @Test
    void testInit() throws Exception {
        login = new Login();
        login.init();
        // Put your assertions here
    }
}
