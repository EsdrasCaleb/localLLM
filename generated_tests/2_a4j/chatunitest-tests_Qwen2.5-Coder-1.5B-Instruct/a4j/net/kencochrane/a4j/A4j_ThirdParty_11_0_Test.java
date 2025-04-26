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

class A4j_ThirdParty_11_0_Test {

    private A4j a4j;

    @BeforeEach
    public void setUp() {
        a4j = new A4j();
    }

    @Test
    public void testThirdParty() {
        // Arrange
        String sellerId = "123";
        String type = "active";
        String page = "1";
        String status = "pending";
        // Act
        SellerSearch result = a4j.ThirdParty(sellerId, type, page, status);
        // Assert
        assertNotNull(result);
        verify(a4j).ThirdParty(anyString(), anyString(), anyString(), anyString());
    }
}
