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
public class A4j_ActorSearch_3_3_Test {

    @InjectMocks
    private A4j a4j;

    @Mock
    private Search search;

    @Test
    public void testActorSearch() {
        String actorName = "actorName";
        String mode = "mode";
        String page = "page";
        ProductInfo productInfo = new ProductInfo();
        when(search.ActorSearch(actorName, mode, page)).thenReturn(productInfo);
        ProductInfo result = a4j.ActorSearch(actorName, mode, page);
        assertEquals(productInfo, result);
        verify(search, times(1)).ActorSearch(actorName, mode, page);
    }
}
