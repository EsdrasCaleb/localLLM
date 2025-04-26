package net.kencochrane.a4j.data;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;

@ExtendWith(MockitoExtension.class)
public class Query_AddToExistingCart_9_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @Test
    public void testAddToExistingCart() {
        String asin = "1234567890";
        String quantity = "2";
        String cartId = "ABC123";
        String hmac = "testHmac";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociatesId&Asin.1234567890=2&CartId=ABC123&Hmac=encodedHmac";
        when(jawsUtil.encodeString(hmac)).thenReturn("encodedHmac");
        String actualUrl = query.AddToExistingCart(asin, quantity, cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
    }
}
