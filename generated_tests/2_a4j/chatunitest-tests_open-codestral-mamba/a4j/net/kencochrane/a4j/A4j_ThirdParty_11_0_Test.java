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
public class A4j_ThirdParty_11_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testThirdParty() {
        String sellerId = "testSellerId";
        String type = "testType";
        String page = "testPage";
        String status = "testStatus";
        // Initialize with expected result
        SellerSearch expectedSellerSearch = new SellerSearch();
        when(search.ThirdParty(sellerId, type, page, status)).thenReturn(expectedSellerSearch);
        SellerSearch actualSellerSearch = a4j.ThirdParty(sellerId, type, page, status);
        assertEquals(expectedSellerSearch, actualSellerSearch);
    }
}
