package com.hf.sfm.util;

import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.SessionFactory;
import org.hibernate.cfg.Configuration;

public class HibernateSessionFactory_closeSession_1_2_Test {

    @Test
    public void testCloseSession() {
        // Arrange
        Session session = mock(Session.class);
        HibernateSessionFactory.threadSession.set(session);
        // Act
        HibernateSessionFactory.closeSession();
        // Assert
        assertDoesNotThrow(() -> HibernateSessionFactory.threadSession.get().close());
    }
}
