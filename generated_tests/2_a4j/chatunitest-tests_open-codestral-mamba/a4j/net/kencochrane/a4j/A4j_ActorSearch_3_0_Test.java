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

public class A4j_ActorSearch_3_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testActorSearch() {
        String actorName = "John Doe";
        String mode = "movie";
        String page = "1";
        // Initialize with expected values
        ProductInfo expected = new ProductInfo();
        when(search.ActorSearch(actorName, mode, page)).thenReturn(expected);
        ProductInfo actual = a4j.ActorSearch(actorName, mode, page);
        assertEquals(expected, actual);
    }
}
