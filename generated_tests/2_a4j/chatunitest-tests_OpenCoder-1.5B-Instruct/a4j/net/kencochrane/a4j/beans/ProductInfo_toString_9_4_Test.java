package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ProductInfo_toString_9_4_Test {

    private ProductInfo productInfo;

    @Test
    public void testToString() throws Exception {
        productInfo = Mockito.mock(ProductInfo.class);
        Method method = ProductInfo.class.getDeclaredMethod("toString");
        method.setAccessible(true);
        Mockito.when(productInfo.getTotalResults()).thenReturn("12345");
        Mockito.when(productInfo.getTotalPages()).thenReturn("10");
        Mockito.when(productInfo.getProductsArrayList()).thenReturn(new ArrayList<>());
        String result = (String) method.invoke(productInfo);
        assertEquals("Total results = 12345\nTotal pages = 10\n# of products = 0\n", result);
    }
}
