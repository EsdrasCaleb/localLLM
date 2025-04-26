package net.kencochrane.a4j.util;

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
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_encodeString_4_0_Test {

    @Test
    void encodeString() {
        a4jUtil util = mock(a4jUtil.class);
        String searchTerm = "Test String";
        when(util.encodeString(searchTerm)).thenReturn("Test%20String");
        assertEquals(util.encodeString(searchTerm), "Test%20String");
    }
}
