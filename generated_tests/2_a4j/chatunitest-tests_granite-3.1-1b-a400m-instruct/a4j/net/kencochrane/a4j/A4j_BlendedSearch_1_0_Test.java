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

class A4j_BlendedSearch_1_0_Test {

    private A4j focal;

    @BeforeEach
    void setUp() {
        focal = new A4j();
    }

    @Test
    void testBlendedSearch() {
        String searchTerm = "test";
        String type = "test";
        BlendedSearch search = focal.BlendedSearch(searchTerm, type);
        // Test case 1: Search term is not found
        BlendedSearch search2 = focal.BlendedSearch("non-existent", type);
        assertEquals(null, search2);
        // Test case 2: Search term is found
        BlendedSearch search3 = focal.BlendedSearch("test", type);
        assertEquals(search, search3);
    }
}
