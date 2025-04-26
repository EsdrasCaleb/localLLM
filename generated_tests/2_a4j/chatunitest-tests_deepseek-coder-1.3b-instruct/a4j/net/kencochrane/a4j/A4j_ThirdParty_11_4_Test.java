package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_ThirdParty_11_4_Test {

    @InjectMocks
    private A4j a4j;

    @Mock
    private Search search;

    @Test
    public void testThirdParty() {
        // Given
        String sellerId = "sellerId";
        String type = "type";
        String page = "page";
        String status = "status";
        SellerSearch expectedSellerSearch = new SellerSearch();
        when(search.ThirdParty(sellerId, type, page, status)).thenReturn(expectedSellerSearch);
        // When
        SellerSearch actualSellerSearch = a4j.ThirdParty(sellerId, type, page, status);
        // Then
        assertEquals(expectedSellerSearch, actualSellerSearch);
    }
}
