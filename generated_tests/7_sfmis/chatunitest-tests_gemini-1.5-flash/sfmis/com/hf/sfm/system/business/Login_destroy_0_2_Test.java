package com.hf.sfm.system.business;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
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

public class Login_destroy_0_2_Test {

    @Test
    void testDestroy() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Login login = new Login();
        Method destroyMethod = Login.class.getDeclaredMethod("destroy");
        // Make the private method accessible for testing
        destroyMethod.setAccessible(true);
        // Call the method and assert that no exception was thrown.  This is a basic test,
        // as the internal workings of destroy() are not specified.  More robust tests would
        // require more information about what "destroy" is supposed to do.
        assertDoesNotThrow(() -> destroyMethod.invoke(login));
    }
}
