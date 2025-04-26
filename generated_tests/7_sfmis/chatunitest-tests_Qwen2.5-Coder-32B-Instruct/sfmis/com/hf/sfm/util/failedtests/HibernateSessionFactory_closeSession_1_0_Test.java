package com.hf.sfm.util;

import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.cfg.Configuration;

public class HibernateSessionFactory_closeSession_1_0_Test {

    @Mock
    private SessionFactory sessionFactoryMock;

    @Mock
    private Session sessionMock;

    @Mock
    private Log logMock;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Replace the static log instance with the mock
        Field logField = HibernateSessionFactory.class.getDeclaredField("log");
        logField.setAccessible(true);
        logField.set(null, logMock);
    }

    @Test
    void testCloseSession_SessionIsNull() {
        // Arrange
        HibernateSessionFactory.threadSession.set(null);
        // Act
        HibernateSessionFactory.closeSession();
        // Assert
        verify(sessionMock, never()).close();
        verify(logMock, never()).debug("关闭session成功！");
        verify(logMock, never()).debug("关闭session失败！");
    }

    @Test
    void testCloseSession_SessionIsNotNullAndCloseSucceeds() {
        // Arrange
        HibernateSessionFactory.threadSession.set(sessionMock);
        doNothing().when(sessionMock).close();
        // Act
        HibernateSessionFactory.closeSession();
        // Assert
        verify(sessionMock).close();
        verify(logMock).debug("关闭session成功！");
        verify(logMock, never()).debug("关闭session失败！");
    }

    @Test
    void testCloseSession_SessionIsNotNullAndCloseFails() {
        // Arrange
        HibernateSessionFactory.threadSession.set(sessionMock);
        doThrow(new HibernateException("Test Exception")).when(sessionMock).close();
        // Act
        HibernateSessionFactory.closeSession();
        // Assert
        verify(sessionMock).close();
        verify(logMock, never()).debug("关闭session成功！");
        verify(logMock).debug("关闭session失败！");
    }
}
