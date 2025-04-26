package net.kencochrane.a4j.util;

import java.net.URLDecoder;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_dencodeString_5_1_Test {

    @Test
    public void dencodeStringTest() {
        // Arrange
        String searchTerm = "search%20term";
        String expectedResult = "search term";
        a4jUtil a4jUtil = new a4jUtil();
        // Act
        String result = a4jUtil.dencodeString(searchTerm);
        // Assert
        assertEquals(expectedResult, result);
    }
}
