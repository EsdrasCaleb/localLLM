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

public class A4j_ThirdParty_11_0_Test {

    @Mock
    private Search searchMock;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testThirdParty() {
        // Arrange
        String sellerId = "seller123";
        String type = "type1";
        String page = "1";
        String status = "active";
        SellerSearch expectedSellerSearch = new SellerSearch();
        when(searchMock.ThirdParty(sellerId, type, page, status)).thenReturn(expectedSellerSearch);
        // Act
        SellerSearch result = a4j.ThirdParty(sellerId, type, page, status);
        // Assert
        assertNotNull(result);
        assertEquals(expectedSellerSearch, result);
        verify(searchMock, times(1)).ThirdParty(sellerId, type, page, status);
    }
}
