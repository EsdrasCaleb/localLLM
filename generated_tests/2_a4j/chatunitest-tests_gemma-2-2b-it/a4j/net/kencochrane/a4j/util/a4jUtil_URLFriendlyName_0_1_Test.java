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

public class a4jUtil_URLFriendlyName_0_1_Test {

    @Test
    void testURLFriendlyName() {
        a4jUtil util = new a4jUtil();
        String expected = "my-name";
        String actual = util.URLFriendlyName("my name");
        assertEquals(expected, actual);
    }
}
