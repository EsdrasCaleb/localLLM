package net.kencochrane.a4j.util;

import java.util.ArrayList;
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

public class a4jUtil_arrayToCommaString_3_0_Test {

    @Test
    public void testArrayToCommaString() {
        a4jUtil a4j = new a4jUtil();
        ArrayList<String> list = new ArrayList<>();
        list.add("test1");
        list.add("test2");
        list.add("test3");
        String result = a4j.arrayToCommaString(list);
        Assertions.assertEquals("test1, test2, test3", result);
    }
}
