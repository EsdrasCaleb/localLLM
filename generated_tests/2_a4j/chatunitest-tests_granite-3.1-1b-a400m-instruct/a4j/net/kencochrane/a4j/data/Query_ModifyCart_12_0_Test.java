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

public class Query_ModifyCart_12_0_Test {

    @Test
    public void testModifyCart() {
        // Given
        Query query = new Query();
        String itemId = "12345";
        String quantity = "2";
        String cartId = "123";
        String hmac = "1234567890abcdef";
        // When
        String modifiedCart = query.ModifyCart(itemId, quantity, cartId, hmac);
        // Then
        assertEquals("http://xml.amazon.com/onca/xml3?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=123&Item.12345=2&CartId=123&Hmac=1234567890abcdef", modifiedCart);
    }
}
