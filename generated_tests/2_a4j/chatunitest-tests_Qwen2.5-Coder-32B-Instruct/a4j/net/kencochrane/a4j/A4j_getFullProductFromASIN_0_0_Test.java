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

public class A4j_getFullProductFromASIN_0_0_Test {

    @Mock
    private Product productMock;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetFullProductFromASIN() {
        // Arrange
        String asin = "B08N5WRWNW";
        String offer = "new";
        String page = "1";
        FullProduct expectedFullProduct = new FullProduct();
        when(productMock.getProduct(asin, offer, page)).thenReturn(expectedFullProduct);
        // Act
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        // Assert
        assertEquals(expectedFullProduct, actualFullProduct);
        verify(productMock, times(1)).getProduct(asin, offer, page);
    }
}
