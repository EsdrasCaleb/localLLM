package net.kencochrane.a4j.data;

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
import java.util.ArrayList;
import java.util.Properties;

class Query_AddToExistingCart_9_4_Test {

    private Query query;

    @BeforeEach
    void setUp() {
        query = new Query();
    }

    @Test
    void testAddToExistingCart() {
        String asin = "9780765317694";
        String quantity = "1";
        String cartId = "123456789";
        String hmac = "A4J1234567890";
        String expected = "http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=[[developer's token goes here]]&t=[associates ID goes here]&Asin.[ASIN goes here]=[quantity goes here]&CartId=[cart ID goes here]&Hmac=[HMAC goes here]";
        String actual = query.AddToExistingCart(asin, quantity, cartId, hmac);
        assertEquals(expected, actual);
    }
}
