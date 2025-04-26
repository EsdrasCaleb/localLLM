package net.kencochrane.a4j.util;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.math.BigDecimal;
import java.util.Optional;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

class a4jUtil_getPrice_2_0_Test {

    @Test
    void getPrice_ValidInput_ReturnsCorrectValue() {
        // Arrange
        a4jUtil a4jUtil = new a4jUtil();
        String input = "123.45";
        // Act
        BigDecimal result = a4jUtil.getPrice(input);
        // Assert
        assertEquals(new BigDecimal("123.45"), result);
    }

    @ParameterizedTest
    @CsvSource({ "123.45, 123.45", "123.456, 123.45", "123, 123.00" })
    void getPrice_ValidInput_RoundsUpToTwoDecimalPlaces(String input, String expected) {
        // Arrange
        a4jUtil a4jUtil = new a4jUtil();
        // Act
        BigDecimal result = a4jUtil.getPrice(input);
        // Assert
        assertEquals(new BigDecimal(expected), result);
    }

    @Test
    void getPrice_InvalidInput_ReturnsZero() {
        // Arrange
        a4jUtil a4jUtil = new a4jUtil();
        String input = "InvalidInput";
        // Act
        BigDecimal result = a4jUtil.getPrice(input);
        // Assert
        assertEquals(new BigDecimal(0.00), result);
    }
}
