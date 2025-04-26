package net.kencochrane.a4j.beans;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SimilarProducts_getProduct_3_0_Test {

    @Test
    void getProduct_negativeIndex_returnsNull() {
        SimilarProducts similarProducts = new SimilarProducts();
        String[] products = { "Product1", "Product2" };
        similarProducts.setProduct(products);
        String product = similarProducts.getProduct(-1);
        assertNull(product);
    }
}
