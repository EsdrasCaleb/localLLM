package com.hf.sfm.util;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.Collectors;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.HashMap;
import java.util.Iterator;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;
import org.hibernate.HibernateException;
import org.hibernate.Query;
import org.hibernate.Session;

public class Loader_run_8_0_Test {

    @Mock
    private BasePara basePara;

    @InjectMocks
    private Loader loader;

    @Test
    public void testRun() {
        // Setup
        MockitoAnnotations.initMocks(this);
        when(basePara.getSqlpath()).thenReturn("sqlfolder/");
        when(basePara.isPaging()).thenReturn(true);
        when(basePara.getStart()).thenReturn(0);
        when(basePara.getLimit()).thenReturn(10);
        when(basePara.getSort()).thenReturn("id");
        when(basePara.getDir()).thenReturn("asc");
        when(basePara.single2plannar()).thenReturn(new String[][] { { "id", "name" } });
        // Call the method to be tested
        loader.run(basePara);
        // Verify the expected behavior
        verify(basePara, times(1)).getSqlpath();
        verify(basePara, times(1)).isPaging();
        verify(basePara, times(1)).getStart();
        verify(basePara, times(1)).getLimit();
        verify(basePara, times(1)).getSort();
        verify(basePara, times(1)).getDir();
        verify(basePara, times(1)).single2plannar();
    }
}
