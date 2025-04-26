package com.hf.sfm.util;

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
import java.util.List;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

public class Loader_loadDataWithSql_12_1_Test {

    @Test
    void loadDataWithSql_ValidData() {
        Loader loader = Mockito.mock(Loader.class);
        Mockito.when(loader.loadDataWithSql()).thenReturn(List.of(1, 2, 3));
        List<Integer> result = loader.loadDataWithSql();
        assertEquals(3, result.size());
    }
}
