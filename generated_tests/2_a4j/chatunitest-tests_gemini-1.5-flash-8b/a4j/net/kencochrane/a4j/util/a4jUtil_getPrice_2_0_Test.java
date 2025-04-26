package net.kencochrane.a4j.util;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.math.BigDecimal;
import java.util.regex.Pattern;
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

    @ParameterizedTest
    @CsvSource({ "123.45,123.45", "12345,12345.00", "123.456,123.46", "abc123.45xyz,123.45", "123.45a,123.45", ".,0.00", ".,0.00", "123.456789,123.46", "1234567890.123456789,1234567890.12", "", "0.00", "0,0.00", "0.0,0.00", "1234.567,1234.57", "-123.45,-123.45", "12345678901234567890.123456789,12345678901234567890.12" })
    void testGetPrice(String inputPrice, String expectedPrice) {
        a4jUtil util = new a4jUtil();
        BigDecimal actualPrice = util.getPrice(inputPrice);
        BigDecimal expectedBigDecimal = new BigDecimal(expectedPrice);
        assertEquals(expectedBigDecimal, actualPrice);
    }

    private String stripString(String allowedString, String price) {
        return price.replaceAll("[^" + allowedString + "]", "");
    }
}
