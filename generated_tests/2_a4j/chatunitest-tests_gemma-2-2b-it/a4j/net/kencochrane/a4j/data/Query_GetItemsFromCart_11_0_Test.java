package net.kencochrane.a4j.data;

import java.util.ArrayList;
import java.util.List;
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

public class Query_GetItemsFromCart_11_0_Test {

    @Test
    void testGetItemsFromCart() {
        Query query = new Query();
        String cartId = "12345";
        String hmac = "someString";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=get&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Hmac=someString";
        String actualUrl = query.GetItemsFromCart(cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
    }
}
