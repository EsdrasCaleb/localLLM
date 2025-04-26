package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductInfo_printProductList_10_4_Test {

    @Test
    public void testPrintProductList() throws Exception {
        ProductInfo productInfo = mock(ProductInfo.class);
        when(productInfo.getTotalResults()).thenReturn("100");
        when(productInfo.getTotalPages()).thenReturn("20");
        when(productInfo.getProductsArrayList()).thenReturn(new ArrayList<>());
        Method methodToTest = productInfo.getClass().getDeclaredMethod("printProductList");
        methodToTest.setAccessible(true);
        String result = (String) methodToTest.invoke(productInfo);
        assertEquals("Total results = 100\nTotal pages = 20\n# of products = 0\n", result);
    }
}
