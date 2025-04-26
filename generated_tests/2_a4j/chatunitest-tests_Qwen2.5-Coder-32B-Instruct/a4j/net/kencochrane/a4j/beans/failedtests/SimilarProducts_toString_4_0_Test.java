package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_toString_4_0_Test {

    private SimilarProducts similarProducts;

    @BeforeEach
    public void setUp() {
        similarProducts = new SimilarProducts();
    }

    @Test
    public void testToString_EmptyList() throws Exception {
        // Given
        ArrayList<String> mockList = new ArrayList<>();
        setSimProducts(similarProducts, mockList);
        // When
        String result = similarProducts.toString();
        // Then
        assertEquals("Similar Products is null or size 0\n", result);
    }

    @Test
    public void testToString_NullList() throws Exception {
        // Given
        setSimProducts(similarProducts, null);
        // When
        String result = similarProducts.toString();
        // Then
        assertEquals("Similar Products is null or size 0\n", result);
    }

    @Test
    public void testToString_SingleProduct() throws Exception {
        // Given
        ArrayList<String> mockList = new ArrayList<>();
        mockList.add("Product1");
        setSimProducts(similarProducts, mockList);
        // When
        String result = similarProducts.toString();
        // Then
        assertEquals("# of Simular products = 1\nProductAction - Product1\n", result);
    }

    @Test
    public void testToString_MultipleProducts() throws Exception {
        // Given
        ArrayList<String> mockList = new ArrayList<>();
        mockList.add("Product1");
        mockList.add("Product2");
        mockList.add("Product3");
        setSimProducts(similarProducts, mockList);
        // When
        String result = similarProducts.toString();
        // Then
        assertEquals("# of Simular products = 3\nProductAction - Product1\nProductAction - Product2\nProductAction - Product3\n", result);
    }

    @Test
    public void testToString_ProductsWithNull() throws Exception {
        // Given
        ArrayList<String> mockList = new ArrayList<>();
        mockList.add("Product1");
        mockList.add(null);
        mockList.add("Product3");
        setSimProducts(similarProducts, mockList);
        // When
        String result = similarProducts.toString();
        // Then
        assertEquals("# of Simular products = 3\nProductAction - Product1\nProductAction - \nProductAction - Product3\n", result);
    }

    private void setSimProducts(SimilarProducts instance, ArrayList<String> value) throws Exception {
        Field field = SimilarProducts.class.getDeclaredField("simProducts");
        field.setAccessible(true);
        field.set(instance, value);
    }
}
