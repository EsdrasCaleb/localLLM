package com.hf.sfm.util;

import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;
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

    @BeforeEach
    public void setUp() {
        mockSession = mock(Session.class);
        HibernateSessionFactory.threadSession.set(mockSession);
    }

    @Test
    public void testCloseSession_NullSession() {
        // Arrange
        HibernateSessionFactory.threadSession.set(null);
        // Act
        HibernateSessionFactory.closeSession();
        // Assert
        assertNull(HibernateSessionFactory.threadSession.get());
        verify(mockSession, times(0)).close();
    }

    @Test
    public void testCloseSession_Failure() {
        // Arrange
        doThrow(new HibernateException("Close failed")).when(mockSession).close();
        // Act
        HibernateSessionFactory.closeSession();
        // Assert
        assertNull(HibernateSessionFactory.threadSession.get());
        verify(mockSession, times(1)).close();
    }
}
