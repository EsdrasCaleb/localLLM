package com.hf.sfm.util;

import static org.junit.Assert.*;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import java.sql.*;
import java.util.logging.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.Session;
import org.hibernate.Transaction;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

public class DaoFactory_commit_2_1_Test {

    private DaoFactory obj;

    private Connection conn;

    private static Log log;

    private Session session;

    private CallableStatement ps;

    private ResultSet rs;

    private Transaction tx;

    @Before
    public void setUp() throws Exception {
        obj = new DaoFactory();
        conn = Mockito.mock(Connection.class);
        log = Mockito.mock(Log.class);
        session = Mockito.mock(Session.class);
        ps = Mockito.mock(CallableStatement.class);
        rs = Mockito.mock(ResultSet.class);
        tx = Mockito.mock(Transaction.class);
    }

    @Test
    public void testCommit() throws Exception {
        obj.commit();
        Mockito.verify(tx).commit();
        Mockito.verify(obj).closeAll();
    }

    @After
    public void tearDown() throws Exception {
        obj = null;
        conn = null;
        log = null;
        session = null;
        ps = null;
        rs = null;
        tx = null;
    }
}
