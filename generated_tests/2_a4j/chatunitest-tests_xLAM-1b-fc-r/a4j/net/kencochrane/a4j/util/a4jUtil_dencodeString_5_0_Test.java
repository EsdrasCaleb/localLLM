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

public class a4jUtil_dencodeString_5_0_Test {

    @Test
    public void dencodeStringTest() {
        a4jUtil util = new a4jUtil();
        String encodedString = "test%20string";
        String expectedDecodedString = "test string";
        String actualDecodedString = util.dencodeString(encodedString);
        assertEquals(expectedDecodedString, actualDecodedString);
    }
}
