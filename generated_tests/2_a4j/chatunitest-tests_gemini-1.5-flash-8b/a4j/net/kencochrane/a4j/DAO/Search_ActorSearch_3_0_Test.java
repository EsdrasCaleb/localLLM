package net.kencochrane.a4j.DAO;

import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.beans.ProductInfo;
import net.kencochrane.a4j.DAO.Search;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;

@ExtendWith(MockitoExtension.class)
class Search_ActorSearch_3_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private Search searchImpl;

    @Test
    void actorSearch_validInput_returnsProductInfo() {
        ProductInfo mockProductInfo = mock(ProductInfo.class);
        when(search.Generic("ActorSearch", "Tom Cruise", "advanced", "lite", "1", "all")).thenReturn(mockProductInfo);
        String actorName = "Tom Cruise";
        String mode = "advanced";
        String page = "1";
        ProductInfo result = searchImpl.ActorSearch(actorName, mode, page);
        assertNotNull(result);
        verify(search).Generic("ActorSearch", actorName, mode, "lite", page, "all");
    }

    @Test
    void actorSearch_nullActorName_throwsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> searchImpl.ActorSearch(null, "advanced", "1"));
    }

    @Test
    void actorSearch_emptyActorName_throwsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> searchImpl.ActorSearch("", "advanced", "1"));
    }

    @Test
    void actorSearch_invalidPage_throwsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> searchImpl.ActorSearch("Tom Cruise", "advanced", "abc"));
    }

    @Test
    void actorSearch_nullMode_throwsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> searchImpl.ActorSearch("Tom Cruise", null, "1"));
    }

    @Test
    void actorSearch_emptyMode_throwsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> searchImpl.ActorSearch("Tom Cruise", "", "1"));
    }

    @Test
    void actorSearch_invalidMode_throwsIllegalArgumentException() {
        assertThrows(IllegalArgumentException.class, () -> searchImpl.ActorSearch("Tom Cruise", "invalidMode", "1"));
    }

    @Test
    void actorSearch_largePageNumber_returnsProductInfo() {
        ProductInfo mockProductInfo = mock(ProductInfo.class);
        when(search.Generic("ActorSearch", "Tom Cruise", "advanced", "lite", "1000", "all")).thenReturn(mockProductInfo);
        String actorName = "Tom Cruise";
        String mode = "advanced";
        String page = "1000";
        ProductInfo result = searchImpl.ActorSearch(actorName, mode, page);
        assertNotNull(result);
        verify(search).Generic("ActorSearch", actorName, mode, "lite", page, "all");
    }

    @Test
    void actorSearch_specialCharacters_returnsProductInfo() {
        ProductInfo mockProductInfo = mock(ProductInfo.class);
        when(search.Generic("ActorSearch", "Tom Cruise%", "advanced", "lite", "1", "all")).thenReturn(mockProductInfo);
        String actorName = "Tom Cruise%";
        String mode = "advanced";
        String page = "1";
        ProductInfo result = searchImpl.ActorSearch(actorName, mode, page);
        assertNotNull(result);
        verify(search).Generic("ActorSearch", actorName, mode, "lite", page, "all");
    }
}
