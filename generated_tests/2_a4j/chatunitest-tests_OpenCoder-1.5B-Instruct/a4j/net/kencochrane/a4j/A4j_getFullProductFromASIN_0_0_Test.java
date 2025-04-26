package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

// Focal class
public class A4j_getFullProductFromASIN_0_0_Test {

    // Test method
    @Test
    public void testGetFullProductFromASIN() throws Exception {
        // Create a mock object
        Product mockProduct = Mockito.mock(Product.class);
        A4j a4j = new A4j();
        // Create a reflection instance of A4j
        A4j a4jInstance = Mockito.spy(a4j);
        // Get the private method
        java.lang.reflect.Method privateMethod = A4j.class.getDeclaredMethod("getFullProductFromASIN", String.class, String.class, String.class);
        privateMethod.setAccessible(true);
        // Invoke the private method and pass mock objects
        FullProduct fullProduct = (FullProduct) privateMethod.invoke(a4jInstance, "asin", "offer", "page");
        // Assertions
        Assertions.assertNotNull(fullProduct);
        // Verify that the method was called
        Mockito.verify(mockProduct).getProduct("asin", "offer", "page");
    }
}
