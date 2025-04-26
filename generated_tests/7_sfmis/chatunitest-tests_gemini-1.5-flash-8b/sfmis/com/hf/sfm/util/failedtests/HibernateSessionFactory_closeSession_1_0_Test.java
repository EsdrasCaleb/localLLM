package com.hf.sfm.util;

import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.cfg.Configuration;

public class HibernateSessionFactory_closeSession_1_0_Test {

    private Session mockSession;

    private HibernateSessionFactory sessionFactory;

    @BeforeEach
    void setUp() {
        mockSession = Mockito.mock(Session.class);
        sessionFactory = new HibernateSessionFactory();
        try {
            Field threadSessionField = HibernateSessionFactory.class.getDeclaredField("threadSession");
            threadSessionField.setAccessible(true);
            // Clear the thread local
            threadSessionField.set(sessionFactory, new ThreadLocal<Session>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        try {
            Field sessionFactoryField = HibernateSessionFactory.class.getDeclaredField("sessionFactory");
            sessionFactoryField.setAccessible(true);
            // Clear the session factory
            sessionFactoryField.set(sessionFactory, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Test
    void closeSessionWithOpenSession() {
        try {
            Field threadSessionField = HibernateSessionFactory.class.getDeclaredField("threadSession");
            threadSessionField.setAccessible(true);
            ((ThreadLocal<Session>) threadSessionField.get(sessionFactory)).set(mockSession);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        doNothing().when(mockSession).close();
        HibernateSessionFactory.closeSession();
        verify(mockSession).close();
        try {
            Field threadSessionField = HibernateSessionFactory.class.getDeclaredField("threadSession");
            threadSessionField.setAccessible(true);
            assertNull(((ThreadLocal<Session>) threadSessionField.get(sessionFactory)).get());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Test
    void closeSessionWithNoOpenSession() {
        HibernateSessionFactory.closeSession();
        try {
            Field threadSessionField = HibernateSessionFactory.class.getDeclaredField("threadSession");
            threadSessionField.setAccessible(true);
            assertNull(((ThreadLocal<Session>) threadSessionField.get(sessionFactory)).get());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Test
    void closeSessionWithException() {
        Session mockSessionWithError = Mockito.mock(Session.class);
        doThrow(new HibernateException("Simulate exception")).when(mockSessionWithError).close();
        try {
            Field threadSessionField = HibernateSessionFactory.class.getDeclaredField("threadSession");
            threadSessionField.setAccessible(true);
            ((ThreadLocal<Session>) threadSessionField.get(sessionFactory)).set(mockSessionWithError);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        HibernateSessionFactory.closeSession();
        // Verify that the exception was caught
    }
}
