package com.hf.sfm.util;

import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.cfg.Configuration;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

@ExtendWith(MockitoExtension.class)
public class HibernateSessionFactory_closeSession_1_1_Test {

    @Mock
    private SessionFactory sessionFactory;

    @InjectMocks
    private HibernateSessionFactory hibernateSessionFactory;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testCloseSession() {
        Session mockSession = mock(Session.class);
        HibernateSessionFactory.threadSession.set(mockSession);
        HibernateSessionFactory.closeSession();
        verify(mockSession, times(1)).close();
        assertNull(HibernateSessionFactory.threadSession.get());
    }

    @Test
    public void testCloseSessionWithNullSession() {
        HibernateSessionFactory.threadSession.set(null);
        HibernateSessionFactory.closeSession();
        assertNull(HibernateSessionFactory.threadSession.get());
    }

    @Test
    public void testCloseSessionWithException() {
        Session mockSession = mock(Session.class);
        doThrow(new HibernateException("Test Exception")).when(mockSession).close();
        HibernateSessionFactory.threadSession.set(mockSession);
        HibernateSessionFactory.closeSession();
        verify(mockSession, times(1)).close();
        assertNull(HibernateSessionFactory.threadSession.get());
    }
}
