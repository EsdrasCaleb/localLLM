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

public class Query_ModifyCart_12_0_Test {

    @Test
    public void testModifyCart() {
        // Given
        Query query = new Query();
        String itemId = "123";
        String quantity = "2";
        String cartId = "456";
        String hmac = "789";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=associatesID&Item." + itemId + "=" + quantity + "&CartId=" + cartId + "&Hmac=" + hmac;
        // When
        String result = query.ModifyCart(itemId, quantity, cartId, hmac);
        // Then
        assertEquals(expectedUrl, result);
    }
}
