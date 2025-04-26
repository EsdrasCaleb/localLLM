package net.kencochrane.a4j.util;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.Properties;

class a4jUtil_arrayToCommaString_3_1_Test {

    @Mock
    private ArrayList<String> mockList;

    @InjectMocks
    private a4jUtil a4jUtil;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testArrayToCommaStringWithEmptyList() {
        assertEquals("", a4jUtil.arrayToCommaString(new ArrayList<>()));
    }

    @Test
    public void testArrayToCommaStringWithSingleElementList() {
        ArrayList<String> list = new ArrayList<>();
        list.add("apple");
        assertEquals("apple", a4jUtil.arrayToCommaString(list));
    }

    @Test
    public void testArrayToCommaStringWithMultipleElementsList() {
        ArrayList<String> list = new ArrayList<>(Arrays.asList("apple", "banana", "cherry"));
        assertEquals("apple, banana, cherry", a4jUtil.arrayToCommaString(list));
    }
}
