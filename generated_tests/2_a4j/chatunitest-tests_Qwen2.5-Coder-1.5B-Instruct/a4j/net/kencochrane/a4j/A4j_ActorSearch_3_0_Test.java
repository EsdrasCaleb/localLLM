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

class A4j_ActorSearch_3_0_Test {

    A4j a4j;

    @BeforeEach
    public void setUp() {
        a4j = new A4j();
    }

    @Test
    public void testActorSearch() throws Exception {
        // Mocking Search class
        Search mockSearch = mock(Search.class);
        // Stubbing the ActorSearch method
        when(mockSearch.ActorSearch("John Doe", "full", "1")).thenReturn(new ProductInfo());
        // Invoking the ActorSearch method
        ProductInfo result = a4j.ActorSearch("John Doe", "full", "1");
        // Verifying the result
        assertNotNull(result);
        verify(mockSearch).ActorSearch("John Doe", "full", "1");
    }
}
