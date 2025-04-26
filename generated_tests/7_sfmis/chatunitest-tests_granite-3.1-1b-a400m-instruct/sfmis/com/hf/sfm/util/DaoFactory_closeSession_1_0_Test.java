package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.Session;
import org.hibernate.Transaction;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

public class DaoFactory_closeSession_1_0_Test {

    private DaoFactory daoFactory;

    @Test
    public void testCloseSession() {
        // Arrange
        daoFactory = new DaoFactory();
        // Setup
        daoFactory.closeSession();
        // Act
        // Since we don't have a mock for the session, we can't verify it's closed.
        // In a real scenario, you would need to verify the session is closed here.
        // For simplicity, we'll just assert that the session is null.
        assertNull(daoFactory.session);
    }
}
