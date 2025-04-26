package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.cfg.Configuration;

public class HibernateSessionFactory_closeSession_1_1_Test {

    @BeforeEach
    public void setUp() {
        HibernateSessionFactory.threadSession.set(null);
    }

    @AfterEach
    public void tearDown() {
        HibernateSessionFactory.closeSession();
    }

    @Test
    public void testCloseSession() {
        Session session = mock(Session.class);
        HibernateSessionFactory.threadSession.set(session);
        HibernateSessionFactory.closeSession();
        verify(session).close();
    }

    @Test
    public void testCloseSessionWithException() {
        Session session = mock(Session.class);
        HibernateSessionFactory.threadSession.set(session);
        doThrow(new HibernateException("Test Exception")).when(session).close();
        HibernateSessionFactory.closeSession();
        verify(session).close();
    }
}
