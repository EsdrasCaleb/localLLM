package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
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

class Query_ModifyCart_12_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "testAssociatesID");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
    }

    @Test
    void testModifyCart() {
        String itemId = "testItemId";
        String quantity = "2";
        String cartId = "testCartId";
        String hmac = "testHmac";
        when(jawsUtil.encodeString(hmac)).thenReturn("encodedHmac");
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociatesID&Item.testItemId=2&CartId=testCartId&Hmac=encodedHmac";
        String actualUrl = query.ModifyCart(itemId, quantity, cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
