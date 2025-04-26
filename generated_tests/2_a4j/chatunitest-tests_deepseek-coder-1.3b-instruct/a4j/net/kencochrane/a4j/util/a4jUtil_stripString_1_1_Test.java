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

public class a4jUtil_stripString_1_1_Test {

    @Test
    public void testStripString() {
        // Arrange
        String allowedChars = "abc";
        String string = "abcdefg";
        a4jUtil a4jUtil = new a4jUtil();
        // Act
        String result = a4jUtil.stripString(allowedChars, string);
        // Assert
        Assertions.assertEquals("", result);
    }
}
