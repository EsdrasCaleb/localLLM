package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Product;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_getFullProductFromASIN_0_0_Test {

    @Mock
    private Product productMock;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetFullProductFromASIN() {
        String asin = "B00005NDO0";
        String offer = "B00005NDO0";
        String page = "B00005NDO0";
        FullProduct mockFullProduct = new FullProduct();
        when(productMock.getProduct(asin, offer, page)).thenReturn(mockFullProduct);
        FullProduct result = a4j.getFullProductFromASIN(asin, offer, page);
        assertEquals(mockFullProduct, result);
    }
}
