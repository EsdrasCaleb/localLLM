package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.cfg.Configuration;

@ExtendWith(MockitoExtension.class)
public class HibernateSessionFactory_closeSession_1_0_Test {

    @Test
    void testCloseSession() {
        HibernateSessionFactory.closeSession();
        // Check the session is null
        assertNull(HibernateSessionFactory.threadSession.get());
    }
}
