package com.hf.sfm.util;

import static org.junit.Assert.assertEquals;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

@RunWith(MockitoJUnitRunner.class)
public class Loader_loadDataWithSql_12_2_Test {

    @Mock
    private Loader loader;

    @Test
    public void testLoadDataWithSql() {
        // Arrange
        List<String[]> pas = new ArrayList<String[]>();
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        pas.add(new String[] { "1", "2", "3" });
        List<Object[]> rs = new ArrayList<Object[]>();
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        rs.add(new Object[] { "1", "2", "3" });
        loader.setTotalCount(10);
        loader.setRs(rs);
        // Act
        loader.getCount();
        // Assert
        assertEquals(10, loader.getTotalCount());
    }
}
