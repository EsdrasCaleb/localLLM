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

class HibernateSessionFactory_closeSession_1_0_Test {

    @Test
    void testCloseSession() {
        assertFalse(HibernateSessionFactory.threadSession.get() != null);
        HibernateSessionFactory.closeSession();
        assertTrue(HibernateSessionFactory.threadSession.get() == null);
    }
}
