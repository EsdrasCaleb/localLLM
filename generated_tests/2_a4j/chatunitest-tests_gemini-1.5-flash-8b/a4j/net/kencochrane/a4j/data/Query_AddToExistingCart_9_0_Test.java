package net.kencochrane.a4j.data;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;

class Query_AddToExistingCart_9_0_Test {

    private Query query;

    private a4jUtil mockA4jUtil;

    @BeforeEach
    void setUp() {
        query = new Query();
        mockA4jUtil = Mockito.mock(a4jUtil.class);
        query.jawsUtil = mockA4jUtil;
        query.serverURL = "http://xml.amazon.com/onca/xml3";
        query.associatesID = "testAssociatesID";
    }

    @Test
    void addToExistingCart_validInput_returnsCorrectURL() {
        String asin = "1234567890";
        String quantity = "1";
        String cartId = "abcdefg";
        String hmac = "testHMAC";
        String expectedURL = "http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociatesID&Asin.1234567890=1&CartId=abcdefg&Hmac=encodedHMAC";
        // Replace with the actual encoded value
        String encodedHMAC = "encodedHMAC";
        Mockito.when(mockA4jUtil.encodeString(hmac)).thenReturn(encodedHMAC);
        String actualURL = query.AddToExistingCart(asin, quantity, cartId, hmac);
        assertEquals(expectedURL, actualURL);
    }

    @Test
    void addToExistingCart_nullInput_returnsEmptyURL() {
        String asin = null;
        String quantity = null;
        String cartId = null;
        String hmac = null;
        String actualURL = query.AddToExistingCart(asin, quantity, cartId, hmac);
        assertEquals("", actualURL);
    }

    @Test
    void addToExistingCart_emptyInput_returnsEmptyURL() {
        String asin = "";
        String quantity = "";
        String cartId = "";
        String hmac = "";
        String actualURL = query.AddToExistingCart(asin, quantity, cartId, hmac);
        assertEquals("", actualURL);
    }
}
