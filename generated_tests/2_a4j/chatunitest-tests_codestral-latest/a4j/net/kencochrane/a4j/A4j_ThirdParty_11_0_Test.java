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

    private String sellerId;

    private String type;

    private String page;

    private String status;

    private SellerSearch sellerSearch;

    @BeforeEach
    public void setUp() {
        sellerId = "123";
        type = "typeA";
        page = "1";
        status = "active";
        sellerSearch = new SellerSearch();
    }

    @Test
    public void testThirdParty() {
        when(search.ThirdParty(sellerId, type, page, status)).thenReturn(sellerSearch);
        SellerSearch result = a4j.ThirdParty(sellerId, type, page, status);
        assertNotNull(result);
        assertEquals(sellerSearch, result);
        verify(search, times(1)).ThirdParty(sellerId, type, page, status);
    }
}
