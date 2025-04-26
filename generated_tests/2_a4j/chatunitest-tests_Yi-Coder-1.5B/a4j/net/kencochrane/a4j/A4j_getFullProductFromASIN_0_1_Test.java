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

@ExtendWith(MockitoExtension.class)
public class A4j_getFullProductFromASIN_0_1_Test {

    // Test class
    @Test
    public void testGetFullProductFromASIN() {
        // Arrange
        A4j a4j = new A4j();
        String asin = "asin";
        String offer = "offer";
        String page = "page";
        // Act
        FullProduct fullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        // Assert
        assertNotNull(fullProduct);
    }
}
