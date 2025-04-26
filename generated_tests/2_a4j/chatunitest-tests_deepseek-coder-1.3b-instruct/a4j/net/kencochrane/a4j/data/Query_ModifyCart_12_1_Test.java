package net.kencochrane.a4j.data;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
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

@RunWith(MockitoJUnitRunner.class)
public class Query_ModifyCart_12_1_Test {

    @Mock
    private a4jUtil mockA4jUtil;

    @InjectMocks
    private Query query;

    @Test
    public void testModifyCart() {
        String itemId = "item1";
        String quantity = "1";
        String cartId = "cart1";
        String hmac = "hmac1";
        StringBuffer buffer = new StringBuffer();
        buffer.append(query.serverURL);
        buffer.append("?");
        buffer.append("ShoppingCart=modify&f=xml&dev-t=");
        buffer.append(query.token);
        buffer.append("&t=");
        buffer.append(query.associatesID);
        buffer.append("&Item.");
        buffer.append(itemId);
        buffer.append("=");
        buffer.append(quantity);
        buffer.append("&CartId=");
        buffer.append(cartId);
        buffer.append("&Hmac=");
        buffer.append(query.jawsUtil.encodeString(hmac));
        when(mockA4jUtil.encodeString(hmac)).thenReturn("encodedHmac");
        String result = query.ModifyCart(itemId, quantity, cartId, hmac);
        verify(mockA4jUtil, times(1)).encodeString(hmac);
        assertEquals(buffer.toString(), result);
    }
}
